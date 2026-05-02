import * as ts from "typescript";
import { LuaTarget } from "../../CompilerOptions";
import * as lua from "../../LuaAST";
import { TransformationContext } from "../context";
import { unsupportedProperty } from "../utils/diagnostics";
import { LuaLibFeature, transformLuaLibFunction } from "../utils/lualib";
import { transformArguments, transformCallAndArguments } from "../visitors/call";
import { expressionResultIsUsed, typeAlwaysHasSomeOfFlags } from "../utils/typescript";
import { moveToPrecedingTemp } from "../visitors/expression-list";
import { isUnpackCall, wrapInTable } from "../utils/lua-ast";

// String-named methods on Array.prototype, derived from the lib.es*.d.ts that
// TSTL is built against (see "lib" in tsconfig.json). A TS dep or lib bump
// that adds a new Array method changes this union and triggers the build-time
// completeness check at the bottom of this block.
type ArrayMethodName = {
    [K in keyof unknown[]]-?: K extends string ? (unknown[][K] extends (...args: any) => any ? K : never) : never;
}[keyof unknown[]];

// Methods that transformArrayPrototypeCall handles below: either by lowering
// to a lualib call, or by listing the name explicitly as known-but-unsupported.
// Maintained by hand. The two Exclude<> assertions afterwards force this list
// to stay in sync with `ArrayMethodName`.
type HandledArrayMethod =
    | "at"
    | "concat"
    | "copyWithin"
    | "entries"
    | "every"
    | "fill"
    | "filter"
    | "find"
    | "findIndex"
    | "findLast"
    | "findLastIndex"
    | "flat"
    | "flatMap"
    | "forEach"
    | "includes"
    | "indexOf"
    | "join"
    | "keys"
    | "lastIndexOf"
    | "map"
    | "pop"
    | "push"
    | "reduce"
    | "reduceRight"
    | "reverse"
    | "shift"
    | "slice"
    | "some"
    | "sort"
    | "splice"
    | "toLocaleString"
    | "toReversed"
    | "toSorted"
    | "toSpliced"
    | "toString"
    | "unshift"
    | "values"
    | "with";

type AssertNever<T extends never> = T;
// If a lib bump adds a method, MissingArrayMethods becomes that name and the
// constraint `extends never` fails. Add the method to HandledArrayMethod and
// either implement it in the switch or list it under the "known but unsupported"
// cases.
export type MissingArrayMethods = AssertNever<Exclude<ArrayMethodName, HandledArrayMethod>>;
// If HandledArrayMethod contains a typo or stale name, ExtraArrayMethods
// surfaces it here.
export type ExtraArrayMethods = AssertNever<Exclude<HandledArrayMethod, ArrayMethodName>>;

export function transformArrayConstructorCall(
    context: TransformationContext,
    node: ts.CallExpression,
    calledMethod: ts.PropertyAccessExpression
): lua.Expression | undefined {
    const signature = context.checker.getResolvedSignature(node);
    const params = transformArguments(context, node.arguments, signature);

    const expressionName = calledMethod.name.text;
    switch (expressionName) {
        case "from":
            return transformLuaLibFunction(context, LuaLibFeature.ArrayFrom, node, ...params);
        case "isArray":
            return transformLuaLibFunction(context, LuaLibFeature.ArrayIsArray, node, ...params);
        case "of":
            return wrapInTable(...params);
        default:
            context.diagnostics.push(unsupportedProperty(calledMethod.name, "Array", expressionName));
    }
}

function createTableLengthExpression(context: TransformationContext, expression: lua.Expression, node?: ts.Expression) {
    if (context.luaTarget === LuaTarget.Lua50) {
        const tableGetn = lua.createTableIndexExpression(
            lua.createIdentifier("table"),
            lua.createStringLiteral("getn")
        );
        return lua.createCallExpression(tableGetn, [expression], node);
    } else {
        return lua.createUnaryExpression(expression, lua.SyntaxKind.LengthOperator, node);
    }
}

/**
 * Optimized single element Array.push
 *
 * array[#array+1] = el
 * return (#array + 1)
 */
function transformSingleElementArrayPush(
    context: TransformationContext,
    node: ts.CallExpression,
    caller: lua.Expression,
    param: lua.Expression
): lua.Expression {
    const arrayIdentifier = lua.isIdentifier(caller) ? caller : moveToPrecedingTemp(context, caller);

    // #array + 1
    let lengthExpression: lua.Expression = lua.createBinaryExpression(
        createTableLengthExpression(context, arrayIdentifier),
        lua.createNumericLiteral(1),
        lua.SyntaxKind.AdditionOperator
    );

    const expressionIsUsed = expressionResultIsUsed(node);
    if (expressionIsUsed) {
        // store length in a temp
        lengthExpression = moveToPrecedingTemp(context, lengthExpression);
    }

    const pushStatement = lua.createAssignmentStatement(
        lua.createTableIndexExpression(arrayIdentifier, lengthExpression),
        param,
        node
    );
    context.addPrecedingStatements(pushStatement);
    return expressionIsUsed ? lengthExpression : lua.createNilLiteral();
}

type ArrayMethodHandler = (
    context: TransformationContext,
    node: ts.CallExpression,
    calledMethod: ts.PropertyAccessExpression,
    caller: lua.Expression,
    params: lua.Expression[]
) => lua.Expression | undefined;

const lualib =
    (feature: LuaLibFeature): ArrayMethodHandler =>
    (context, node, _calledMethod, caller, params) =>
        transformLuaLibFunction(context, feature, node, caller, ...params);

const unsupported: ArrayMethodHandler = (context, _node, calledMethod) => {
    context.diagnostics.push(unsupportedProperty(calledMethod.name, "array", calledMethod.name.text));
    return undefined;
};

// Dispatch table; the Record<HandledArrayMethod, ...> type forces an entry
// for every handled method, so omitting one fails TSTL build. Combined with
// the MissingArrayMethods/ExtraArrayMethods checks above, every drift axis
// (lib ↔ HandledArrayMethod ↔ dispatch) is covered at TS check time.
const arrayMethodHandlers: { [K in HandledArrayMethod]: ArrayMethodHandler } = {
    at: lualib(LuaLibFeature.ArrayAt),
    concat: lualib(LuaLibFeature.ArrayConcat),
    entries: lualib(LuaLibFeature.ArrayEntries),
    fill: lualib(LuaLibFeature.ArrayFill),
    push(context, node, _calledMethod, caller, params) {
        if (node.arguments.length === 1) {
            const param = params[0] ?? lua.createNilLiteral();
            if (isUnpackCall(param)) {
                return transformLuaLibFunction(
                    context,
                    LuaLibFeature.ArrayPushArray,
                    node,
                    caller,
                    (param as lua.CallExpression).params[0] ?? lua.createNilLiteral()
                );
            }
            if (!lua.isDotsLiteral(param)) {
                return transformSingleElementArrayPush(context, node, caller, param);
            }
        }
        return transformLuaLibFunction(context, LuaLibFeature.ArrayPush, node, caller, ...params);
    },
    reverse: lualib(LuaLibFeature.ArrayReverse),
    shift: (_context, node, _calledMethod, caller) =>
        lua.createCallExpression(
            lua.createTableIndexExpression(lua.createIdentifier("table"), lua.createStringLiteral("remove")),
            [caller, lua.createNumericLiteral(1)],
            node
        ),
    unshift: lualib(LuaLibFeature.ArrayUnshift),
    sort: lualib(LuaLibFeature.ArraySort),
    pop: (_context, node, _calledMethod, caller) =>
        lua.createCallExpression(
            lua.createTableIndexExpression(lua.createIdentifier("table"), lua.createStringLiteral("remove")),
            [caller],
            node
        ),
    forEach: lualib(LuaLibFeature.ArrayForEach),
    find: lualib(LuaLibFeature.ArrayFind),
    findIndex: lualib(LuaLibFeature.ArrayFindIndex),
    includes: lualib(LuaLibFeature.ArrayIncludes),
    indexOf: lualib(LuaLibFeature.ArrayIndexOf),
    map: lualib(LuaLibFeature.ArrayMap),
    filter: lualib(LuaLibFeature.ArrayFilter),
    reduce: lualib(LuaLibFeature.ArrayReduce),
    reduceRight: lualib(LuaLibFeature.ArrayReduceRight),
    some: lualib(LuaLibFeature.ArraySome),
    every: lualib(LuaLibFeature.ArrayEvery),
    slice: lualib(LuaLibFeature.ArraySlice),
    splice: lualib(LuaLibFeature.ArraySplice),
    join(context, node, calledMethod, caller, params) {
        const callerType = context.checker.getTypeAtLocation(calledMethod.expression);
        const elementType = context.checker.getElementTypeOfArrayType(callerType);
        if (
            elementType &&
            typeAlwaysHasSomeOfFlags(context, elementType, ts.TypeFlags.StringLike | ts.TypeFlags.NumberLike)
        ) {
            const defaultSeparatorLiteral = lua.createStringLiteral(",");
            const param = params[0] ?? lua.createNilLiteral();
            const parameters = [
                caller,
                node.arguments.length === 0
                    ? defaultSeparatorLiteral
                    : lua.isStringLiteral(param)
                    ? param
                    : lua.createBinaryExpression(param, defaultSeparatorLiteral, lua.SyntaxKind.OrOperator),
            ];

            return lua.createCallExpression(
                lua.createTableIndexExpression(lua.createIdentifier("table"), lua.createStringLiteral("concat")),
                parameters,
                node
            );
        }
        return transformLuaLibFunction(context, LuaLibFeature.ArrayJoin, node, caller, ...params);
    },
    flat: lualib(LuaLibFeature.ArrayFlat),
    flatMap: lualib(LuaLibFeature.ArrayFlatMap),
    toReversed: lualib(LuaLibFeature.ArrayToReversed),
    toSorted: lualib(LuaLibFeature.ArrayToSorted),
    toSpliced: lualib(LuaLibFeature.ArrayToSpliced),
    with: lualib(LuaLibFeature.ArrayWith),

    // Known but not lowered: emit the existing diagnostic.
    copyWithin: unsupported,
    findLast: unsupported,
    findLastIndex: unsupported,
    keys: unsupported,
    lastIndexOf: unsupported,
    toLocaleString: unsupported,
    toString: unsupported,
    values: unsupported,
};

export function transformArrayPrototypeCall(
    context: TransformationContext,
    node: ts.CallExpression,
    calledMethod: ts.PropertyAccessExpression
): lua.Expression | undefined {
    const signature = context.checker.getResolvedSignature(node);
    const [caller, params] = transformCallAndArguments(context, calledMethod.expression, node.arguments, signature);

    // Index lookup naturally returns undefined for unknown names (e.g. user on
    // a newer lib than TSTL was built against), so no separate membership
    // check is needed.
    const handler = (arrayMethodHandlers as Record<string, ArrayMethodHandler | undefined>)[calledMethod.name.text];
    if (!handler) {
        context.diagnostics.push(unsupportedProperty(calledMethod.name, "array", calledMethod.name.text));
        return undefined;
    }
    return handler(context, node, calledMethod, caller, params);
}

// String-named, non-method properties on Array.prototype, derived from the
// lib.es*.d.ts that TSTL is built against. Today this is just `length`; the
// pattern is here so a future lib addition (or a TSTL contributor wiring up
// something new like a `Symbol.toStringTag`-equivalent string property)
// surfaces at TS check time the same way method drift does.
type ArrayPropertyName = {
    [K in keyof unknown[]]-?: K extends string ? (unknown[][K] extends (...args: any) => any ? never : K) : never;
}[keyof unknown[]];

type HandledArrayProperty = "length";

export type MissingArrayProperties = AssertNever<Exclude<ArrayPropertyName, HandledArrayProperty>>;
export type ExtraArrayProperties = AssertNever<Exclude<HandledArrayProperty, ArrayPropertyName>>;

type ArrayPropertyHandler = (
    context: TransformationContext,
    node: ts.PropertyAccessExpression
) => lua.Expression | undefined;

const arrayPropertyHandlers: { [K in HandledArrayProperty]: ArrayPropertyHandler } = {
    length: (context, node) => createTableLengthExpression(context, context.transformExpression(node.expression), node),
};

export function transformArrayProperty(
    context: TransformationContext,
    node: ts.PropertyAccessExpression
): lua.Expression | undefined {
    const handler = (arrayPropertyHandlers as Record<string, ArrayPropertyHandler | undefined>)[node.name.text];
    return handler?.(context, node);
}
